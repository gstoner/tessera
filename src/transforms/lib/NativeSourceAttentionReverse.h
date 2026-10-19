// Native source-attention reverse outlining. Clone paired AD SSA; never
// manufacture derivative expressions or physical kernels at this boundary.
#pragma once
#include "NativeAttentionRowProjection.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "../../compiler/ir/AttentionADContract.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Base64.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/SHA256.h"
#include <cmath>
#include <functional>

namespace tessera::source_attention_reverse {
using namespace mlir;
inline std::string print(ModuleOp module) {
  std::string text; llvm::raw_string_ostream stream(text);
  module.print(stream); stream.flush(); return text + "\n";
}
inline std::string hash(llvm::StringRef text) {
  return llvm::toHex(llvm::SHA256::hash(llvm::arrayRefFromStringRef(text)), true);
}
inline bool staticTensor(Type type) {
  auto tensor = dyn_cast_or_null<RankedTensorType>(type);
  if (!tensor || tensor.getEncoding() || !tensor.hasStaticShape() ||
      (!tensor.getElementType().isF16() && !tensor.getElementType().isBF16() &&
       !tensor.getElementType().isF32())) return false;
  int64_t bytes = tensor.getElementTypeBitWidth() / 8;
  for (int64_t extent : tensor.getShape()) {
    if (extent <= 0 || extent > INT32_MAX || bytes > INT64_MAX / extent) return false;
    bytes *= extent;
  }
  return true;
}
inline DictionaryAttr normAttributes(Operation *op) {
  NamedAttrList fields(op->getAttrs()); fields.erase("tessera.autodiff.activity");
  return fields.getDictionary(op->getContext());
}
inline LogicalResult preflight(ModuleOp module) {
  auto reject = [&]() { return module.emitError(
      "native source attention reverse requires a static homogeneous-half reverse-marked RMSNorm to saved-LSE attention source"); };
  auto target=module->getAttrOfType<StringAttr>("tessera.target");
  auto arch=module->getAttrOfType<StringAttr>("tessera.arch");
  SmallVector<func::FuncOp> functions(module.getOps<func::FuncOp>());
  if (!target || target.getValue()!="nvidia_sm120" || !arch || arch.getValue()!="sm_120" ||
      functions.size()!=1 || llvm::range_size(module.getBody()->getOperations())!=1 ||
      module->hasAttr("tessera.native.source_attention_reverse_program_json") ||
      module->hasAttr("tessera.attention_shape_bounds")) return reject();
  auto sequenceRaw=module->getAttr("tessera.attention_sequence_bounds");
  auto sequence=dyn_cast_or_null<DenseI64ArrayAttr>(sequenceRaw);
  if (sequenceRaw && (!sequence || sequence.size()!=2 || sequence[0]<=0 ||
      sequence[1]<=0 || sequence[0]>INT32_MAX || sequence[1]>INT32_MAX))
    return reject();
  for (StringRef request:{"tessera.native.source_attention_reverse_rows_request",
                          "tessera.native.source_attention_reverse_auto_request"})
    if (auto marker=module->getAttr(request); marker && !isa<UnitAttr>(marker)) return reject();
  auto fn=functions.front();
  auto mode=fn->getAttrOfType<StringAttr>("tessera.autodiff");
  if (!mode || mode.getValue()!="reverse" || !fn.getBody().hasOneBlock() ||
      fn.getNumArguments()!=3 || fn.getNumResults()!=2) return reject();
  SmallVector<Operation *> body;
  for (auto &op:fn.getBody().front().without_terminator()) body.push_back(&op);
  if (body.size()!=2 || body[0]->getName().getStringRef()!="tessera.rmsnorm" ||
      body[1]->getName().getStringRef()!="tessera.flash_attn") return reject();
  auto norm=body[0], attention=body[1];
  if (norm->getNumOperands()!=1 || norm->getNumResults()!=1 ||
      attention->getOperand(0)!=norm->getResult(0) ||
      !denseAttentionAD(attention,false,true,false,true) || attention->getNumResults()!=2)
    return reject();
  auto q=dyn_cast<BlockArgument>(norm->getOperand(0));
  auto k=dyn_cast<BlockArgument>(attention->getOperand(1));
  auto v=dyn_cast<BlockArgument>(attention->getOperand(2));
  if (!q || !k || !v || q.getOwner()!=&fn.getBody().front() || k.getOwner()!=q.getOwner() ||
      v.getOwner()!=q.getOwner() || q==k || q==v || k==v) return reject();
  auto qt=dyn_cast<RankedTensorType>(q.getType());
  if (!qt || (!qt.getElementType().isF16() && !qt.getElementType().isBF16()) ||
      norm->getResult(0).getType()!=qt) return reject();
  for (unsigned i=0;i<3;++i) {
    auto type=dyn_cast<RankedTensorType>(fn.getArgumentTypes()[i]);
    if (!staticTensor(type) || type.getRank()!=4 || type.getElementType()!=qt.getElementType() ||
        (fn.getArgAttrDict(i) && !fn.getArgAttrDict(i).empty())) return reject();
  }
  if (sequence) {
    auto kt=cast<RankedTensorType>(k.getType());
    auto vt=cast<RankedTensorType>(v.getType());
    if (qt.getDimSize(2)>sequence[0] || kt.getDimSize(2)>sequence[1] ||
        qt.getDimSize(0)>INT32_MAX/qt.getDimSize(1) ||
        qt.getDimSize(0)*qt.getDimSize(1)>INT32_MAX/sequence[0])
      return reject();
    auto capacities=DenseI64ArrayAttr::get(module.getContext(),
        {qt.getDimSize(0),qt.getDimSize(1),kt.getDimSize(1),sequence[0],
         sequence[1],qt.getDimSize(3),vt.getDimSize(3)});
    SmallVector<int64_t> symbolic{qt.getDimSize(0),qt.getDimSize(1),
        kt.getDimSize(1),ShapedType::kDynamic,ShapedType::kDynamic,
        qt.getDimSize(3),vt.getDimSize(3)};
    if (failed(resolveNativeAttentionShape(attention,symbolic,capacities)))
      return failure();
  }
  int64_t rows=1;
  for (int64_t extent:qt.getShape().drop_back()) {
    if (rows>INT32_MAX/extent) return reject(); rows*=extent;
  }
  for (Type type:attention->getResultTypes()) if (!staticTensor(type)) return reject();
  auto output=cast<RankedTensorType>(attention->getResult(0).getType());
  if (output.getElementType()!=qt.getElementType()) return reject();
  auto ret=dyn_cast<func::ReturnOp>(fn.getBody().front().getTerminator());
  if (!ret || ret.getNumOperands()!=2 || ret.getOperand(0)==ret.getOperand(1) ||
      !llvm::all_of(ret.getOperands(),[&](Value value) {
        return value==attention->getResult(0) || value==attention->getResult(1);
      })) return reject();
  auto policy=norm->getAttrOfType<DictionaryAttr>("numeric_policy");
  auto accum=policy?policy.getAs<StringAttr>("accum"):StringAttr{};
  if (!accum || accum.getValue()!="fp32" || !isMemoryEffectFree(norm)) return reject();
  for (auto field:policy) {
    if (field.getName()=="accum") continue;
    auto spelling=dyn_cast<StringAttr>(field.getValue());
    if (field.getName()!="storage" || !spelling || spelling.getValue()!=
        (qt.getElementType().isF16()?"fp16":"bf16")) return reject();
  }
  for (auto field:norm->getAttrs()) {
    auto key=field.getName().getValue();
    if (key!="eps" && key!="axis" && key!="numeric_policy" &&
        key!="tessera.effect_kind" && key!="tessera.autodiff.activity") return reject();
  }
  if (auto axis=norm->getAttrOfType<IntegerAttr>("axis"); axis && axis.getInt()!=-1 && axis.getInt()!=3)
    return reject();
  if (auto eps=norm->getAttrOfType<FloatAttr>("eps"); eps &&
      (!std::isfinite(float(eps.getValueAsDouble())) || float(eps.getValueAsDouble())<=0)) return reject();
  if (auto requested=fn->getAttr("tessera.autodiff.wrt_indices")) {
    auto indices=dyn_cast<ArrayAttr>(requested); llvm::SmallDenseSet<int64_t> seen;
    if (!indices || indices.empty()) return reject();
    for (auto raw:indices) {
      auto index=dyn_cast<IntegerAttr>(raw);
      if (!index || !index.getType().isInteger(64) || index.getInt()<0 || index.getInt()>2 ||
          !seen.insert(index.getInt()).second) return reject();
    }
  }
  return success();
}
// Every non-scalar tensor in this isolated source program has its sequence
// role at axis 2. B/H/D remain fixed; Sq/Sk identity is explicit in call and
// buffer roles, never inferred by equal trace extents.
inline LogicalResult projectBoundedProgram(ModuleOp program, StringRef rootName,
    DenseI64ArrayAttr bounds, bool rows, llvm::json::Array &graphs) {
  OpBuilder builder(program.getContext());
  program->setAttr("tessera.attention_shape_bounds",bounds);
  auto project=[](Type raw) -> Type {
    auto type=dyn_cast<RankedTensorType>(raw);
    if (!type || type.getRank()<3) return raw;
    SmallVector<int64_t> dims(type.getShape());
    dims[2]=ShapedType::kDynamic;
    return RankedTensorType::get(dims,type.getElementType(),type.getEncoding());
  };
  for (auto fn:program.getOps<func::FuncOp>()) {
    for (auto argument:fn.getArguments()) argument.setType(project(argument.getType()));
    fn.walk([&](Operation *op) {
      for (auto result:op->getResults()) result.setType(project(result.getType()));
    });
    auto ret=cast<func::ReturnOp>(fn.getBody().front().getTerminator());
    fn.setFunctionType(builder.getFunctionType(fn.getBody().front().getArgumentTypes(),ret.getOperandTypes()));
    if (fn.getSymName()==rootName) continue;
    if (fn.getSymName()==(rootName+"__member_0").str())
      fn->setAttr("tessera.native.query_sequence_bound",builder.getI64IntegerAttr(bounds[3]));
    if (fn->hasAttr("tessera.native.sm120_row_producer")) {
      auto type=cast<RankedTensorType>(fn.getArgumentTypes()[0]);
      fn->setAttr("tessera.native.sm120_row_producer",builder.getDictionaryAttr({
        builder.getNamedAttr("schema",builder.getI64IntegerAttr(8)),
        builder.getNamedAttr("rows",builder.getI64IntegerAttr(bounds[0]*bounds[1]*bounds[3])),
        builder.getNamedAttr("columns",builder.getI64IntegerAttr(type.getDimSize(3))),
        builder.getNamedAttr("sequence_bound",builder.getI64IntegerAttr(bounds[3])),
        builder.getNamedAttr("storage",builder.getStringAttr(type.getElementType().isF16()?"f16":"bf16")),
        builder.getNamedAttr("accum",builder.getStringAttr("f32")),
        builder.getNamedAttr("body_hash",builder.getStringAttr(nativeSM120RowBodyHash(fn)))}));
    }
  }
  if (failed(verify(program))) return failure();
  graphs.clear();
  for (unsigned i=0;i<(rows?5u:4u);++i) {
    auto fn=program.lookupSymbol<func::FuncOp>((rootName+"__member_"+Twine(i)).str());
    if (!fn) return program.emitError("bounded source member role was lost");
    auto child=ModuleOp::create(program.getLoc());
    child->setAttrs(program->getAttrs());
    auto copy=cast<func::FuncOp>(fn->clone());copy.setPublic();child.getBody()->push_back(copy);
    if (failed(verify(child))) {child.erase();return failure();}
    graphs.push_back(print(child));child.erase();
  }
  return success();
}
inline LogicalResult emit(ModuleOp module,llvm::StringRef originalSource) {
  auto reject=[&]() { return module.emitError(
      "native source attention reverse outline lost its paired SSA, saved-state roles or normalization lineage"); };
  auto target=module->getAttrOfType<StringAttr>("tessera.target");
  auto arch=module->getAttrOfType<StringAttr>("tessera.arch");
  SmallVector<func::FuncOp> functions(module.getOps<func::FuncOp>());
  func::FuncOp backward;
  for (auto fn:functions) {
    auto role=fn->getAttrOfType<StringAttr>("tessera.autodiff.role");
    if (role && role.getValue()=="backward") backward=fn;
  }
  if (!target || target.getValue()!="nvidia_sm120" || !arch || arch.getValue()!="sm_120" ||
      functions.size()!=2 || !backward || !backward.getBody().hasOneBlock()) return reject();
  auto forwardRef=backward->getAttrOfType<FlatSymbolRefAttr>("tessera.autodiff.forward");
  auto forward=forwardRef?module.lookupSymbol<func::FuncOp>(forwardRef.getValue()):func::FuncOp{};
  if (!forward || forward==backward || !forward.getBody().hasOneBlock() ||
      forward.getNumArguments()!=3 || forward.getNumResults()!=4 ||
      backward.getNumArguments()!=7 || backward.getNumResults()!=3) return reject();
  SmallVector<Operation *> primal;
  for (auto &op:forward.getBody().front().without_terminator()) primal.push_back(&op);
  if (primal.size()!=2 || primal[0]->getName().getStringRef()!="tessera.rmsnorm" ||
      primal[1]->getName().getStringRef()!="tessera_attn.checkpoint_forward") return reject();
  auto norm=primal[0], attention=primal[1];
  if (norm->getNumOperands()!=1 || norm->getNumResults()!=1 ||
      attention->getNumOperands()!=3 || attention->getNumResults()!=2 ||
      attention->getOperand(0)!=norm->getResult(0)) return reject();
  auto q=dyn_cast<BlockArgument>(norm->getOperand(0));
  auto k=dyn_cast<BlockArgument>(attention->getOperand(1));
  auto v=dyn_cast<BlockArgument>(attention->getOperand(2));
  if (!q || !k || !v || q.getOwner()!=&forward.getBody().front() ||
      k.getOwner()!=q.getOwner() || v.getOwner()!=q.getOwner() || q==k || q==v || k==v)
    return reject();
  SmallVector<int64_t> roles{int64_t(q.getArgNumber()),int64_t(k.getArgNumber()),int64_t(v.getArgNumber())};
  auto forwardReturn=dyn_cast<func::ReturnOp>(forward.getBody().front().getTerminator());
  auto backwardReturn=dyn_cast<func::ReturnOp>(backward.getBody().front().getTerminator());
  if (!forwardReturn || !backwardReturn || forwardReturn.getNumOperands()!=4 ||
      backwardReturn.getNumOperands()!=3 || forwardReturn.getOperand(2)!=attention->getResult(0) ||
      forwardReturn.getOperand(3)!=attention->getResult(1)) return reject();
  if (forwardReturn.getOperand(0)==forwardReturn.getOperand(1) ||
      !llvm::all_of(forwardReturn.getOperands().take_front(2),[&](Value value) {
        return value==attention->getResult(0) || value==attention->getResult(1);
      })) return reject();
  int64_t outputSeed=forwardReturn.getOperand(0)==attention->getResult(0)?3:4;
  int64_t lseSeed=outputSeed==3?4:3;
  for (unsigned i=0;i<3;++i)
    if (backward.getArgument(i).getType()!=forward.getArgument(i).getType()) return reject();
  for (unsigned i=0;i<4;++i)
    if (backward.getArgument(3+i).getType()!=forward.getResultTypes()[i]) return reject();
  Operation *product=nullptr;
  for (auto &op:backward.getBody().front().without_terminator())
    if (op.getName().getStringRef()=="tessera_attn.checkpoint_backward") {
      if (product) return reject(); product=&op;
    }
  if (!product || product->getNumOperands()!=7 || product->getNumResults()!=3 ||
      product->getOperand(0)!=backward.getArgument(outputSeed) ||
      product->getOperand(2)!=backward.getArgument(roles[1]) ||
      product->getOperand(3)!=backward.getArgument(roles[2]) ||
      product->getOperand(4)!=backward.getArgument(5) ||
      product->getOperand(5)!=backward.getArgument(6) ||
      product->getOperand(6)!=backward.getArgument(lseSeed) ||
      product->getAttrOfType<BoolAttr>("lse_cotangent")!=BoolAttr::get(module.getContext(),true)) return reject();
  auto recomputed=product->getOperand(1).getDefiningOp();
  if (!recomputed || recomputed->getName()!=norm->getName() || recomputed->getNumOperands()!=1 ||
      recomputed->getNumResults()!=1 || recomputed->getOperand(0)!=backward.getArgument(roles[0]) ||
      recomputed->getResultTypes()!=norm->getResultTypes() || normAttributes(recomputed)!=normAttributes(norm) ||
      !isMemoryEffectFree(recomputed)) return reject();
  if (product->getAttr("scale")!=attention->getAttr("scale") ||
      product->getAttr("causal")!=attention->getAttr("causal") ||
      backwardReturn.getOperand(roles[1])!=product->getResult(1) ||
      backwardReturn.getOperand(roles[2])!=product->getResult(2)) return reject();
  Value sourceRoot=backward.getArgument(roles[0]), queryGradient=product->getResult(0);
  Value sourceGradient=backwardReturn.getOperand(roles[0]);
  llvm::SmallPtrSet<Operation *,32> live; bool usedRoot=false,usedGradient=false;
  std::function<bool(Value)> visit=[&](Value value) {
    if (value==sourceRoot) {usedRoot=true;return true;}
    if (value==queryGradient) {usedGradient=true;return true;}
    auto op=value.getDefiningOp();
    if (!op || op->getBlock()!=&backward.getBody().front() || op->getNumRegions() ||
        !isMemoryEffectFree(op)) return false;
    if (!live.insert(op).second) return true;
    auto name=op->getName().getStringRef();
    if (name!="tessera.cast" && name!="tessera.normalization_stats" &&
        name!="tessera.broadcast_in_dim" && name!="tessera.mul" && name!="tessera.reduce" &&
        name!="tessera.sub" && name!="tessera.add" && name!="arith.constant") return false;
    return llvm::all_of(op->getOperands(),[&](Value operand){return visit(operand);});
  };
  if (!visit(sourceGradient) || !usedRoot || !usedGradient || sourceGradient.getType()!=q.getType()) return reject();
  std::string paired=print(module);
  OpBuilder builder(module.getContext());
  std::string rootName=(forward.getSymName()+"__source_reverse").str();
  if (module.lookupSymbol(rootName)) return reject();
  auto sequence=module->getAttrOfType<DenseI64ArrayAttr>("tessera.attention_sequence_bounds");
  DenseI64ArrayAttr bounds;
  if (sequence) {
    auto qt=cast<RankedTensorType>(q.getType()),kt=cast<RankedTensorType>(k.getType()),
         vt=cast<RankedTensorType>(v.getType());
    bounds=builder.getDenseI64ArrayAttr({qt.getDimSize(0),qt.getDimSize(1),
        kt.getDimSize(1),sequence[0],sequence[1],qt.getDimSize(3),vt.getDimSize(3)});
  }
  bool rows=module->hasAttr("tessera.native.source_attention_reverse_rows_request");
  if (!rows && module->hasAttr("tessera.native.source_attention_reverse_auto_request")) {
    // Named SM120 static-half source policy, measured against the inline route.
    // Use the actual normalization input, not a frontend argument position.
    auto queryType=cast<RankedTensorType>(q.getType());
    auto keyType=cast<RankedTensorType>(k.getType());
    auto valueType=cast<RankedTensorType>(v.getType());
    int64_t logicalRows=1;
    for (int64_t extent:queryType.getShape().drop_back()) logicalRows*=extent;
    if (bounds) logicalRows=bounds[0]*bounds[1]*bounds[3];
    rows=logicalRows>=32 && queryType.getDimSize(3)>=32 &&
         (bounds?bounds[4]:keyType.getDimSize(2))>=4 && valueType.getDimSize(3)>=4;
  }
  unsigned memberCount=rows?5:4, pullbackIndex=memberCount-1;
  SmallVector<func::FuncOp> members; llvm::json::Array graphs;
  for (unsigned index=0;index<memberCount;++index) {
    Operation *op=index==0?norm:index==1?attention:product;
    SmallVector<Type> inputs=index==pullbackIndex?SmallVector<Type>{sourceRoot.getType(),queryGradient.getType()}:
        SmallVector<Type>(op->getOperandTypes());
    SmallVector<Type> results=index==pullbackIndex?SmallVector<Type>{sourceGradient.getType()}:
        SmallVector<Type>(op->getResultTypes());
    if (rows && (index==2 || index==3)) {
      bool consumer=index==3;
      inputs.clear();
      for (auto [operandIndex, operand]:llvm::enumerate(product->getOperands())) {
        if (operandIndex==4 || (consumer && operandIndex==6)) continue;
        inputs.push_back(operand.getType());
      }
      Type rowType=product->getOperand(5).getType();
      if (consumer) inputs.push_back(rowType);
      else results={rowType};
    }
    builder.setInsertionPointToEnd(module.getBody());
    std::string symbol=rootName+"__member_"+std::to_string(index);
    if (module.lookupSymbol(symbol)) return reject();
    auto member=func::FuncOp::create(builder,module.getLoc(),symbol,builder.getFunctionType(inputs,results));
    member.setPrivate(); auto block=member.addEntryBlock(); OpBuilder body(block,block->begin()); IRMapping map;
    if (index==pullbackIndex) {
      map.map(sourceRoot,block->getArgument(0)); map.map(queryGradient,block->getArgument(1));
      for (auto &original:backward.getBody().front().without_terminator())
        if (live.contains(&original)) {
          auto clone=body.clone(original,map);clone->removeAttr("tessera.autodiff.activity");
        }
      func::ReturnOp::create(body,module.getLoc(),map.lookup(sourceGradient));
    } else if (rows && (index==2 || index==3)) {
      auto clone=tessera::attention_rows::createProjection(
          body,product,block->getArguments(),results,index==3);
      func::ReturnOp::create(body,module.getLoc(),clone->getResults());
    } else {
      for (auto [operand,argument]:llvm::zip(op->getOperands(),block->getArguments())) map.map(operand,argument);
      auto clone=body.clone(*op,map);clone->removeAttr("tessera.autodiff.activity");
      func::ReturnOp::create(body,module.getLoc(),clone->getResults());
    }
    SmallVector<Attribute> arguments,outputs;
    for (unsigned i=0;i<inputs.size();++i) arguments.push_back(builder.getStringAttr("input_"+std::to_string(i)));
    for (unsigned i=0;i<results.size();++i) outputs.push_back(builder.getStringAttr("output_"+std::to_string(i)));
    member->setAttr("tessera.argument_bindings",builder.getArrayAttr(arguments));
    member->setAttr("tessera.result_bindings",builder.getArrayAttr(outputs));
    if (rows && (index==2 || index==3)) {
      member->setAttr(index==3?"tessera.checkpoint_gradient_activity":
          "tessera.row_correction_gradient_activity",builder.getDenseI64ArrayAttr({1,1,1}));
      if (index==3) {
        member->setAttr("tessera.checkpoint_gradient_output",builder.getStringAttr("compact_v1"));
        member->setAttr("tessera.checkpoint_gradient_launch",builder.getStringAttr("packed_v1"));
        member->setAttr("tessera.checkpoint_gradient_threads",builder.getI64IntegerAttr(128));
      }
    }
    if (index==pullbackIndex) {
      auto type=cast<RankedTensorType>(sourceRoot.getType()); int64_t rows=1;
      for (int64_t extent:type.getShape().drop_back()) rows*=extent;
      member->setAttr("tessera.native.sm120_row_producer",builder.getDictionaryAttr({
        builder.getNamedAttr("schema",builder.getI64IntegerAttr(6)),
        builder.getNamedAttr("rows",builder.getI64IntegerAttr(rows)),
        builder.getNamedAttr("columns",builder.getI64IntegerAttr(type.getShape().back())),
        builder.getNamedAttr("storage",builder.getStringAttr(type.getElementType().isF16()?"f16":"bf16")),
        builder.getNamedAttr("accum",builder.getStringAttr("f32")),
        builder.getNamedAttr("body_hash",builder.getStringAttr(nativeSM120RowBodyHash(member)))}));
    }
    if (failed(verify(member))) return failure(); members.push_back(member);
    auto child=ModuleOp::create(module.getLoc());
    child->setAttr("tessera.target",target);child->setAttr("tessera.arch",arch);
    auto copy=cast<func::FuncOp>(member->clone());copy.setPublic();child.getBody()->push_back(copy);
    if (failed(verify(child))) {child.erase();return failure();}
    graphs.push_back(print(child));child.erase();
  }
  SmallVector<int64_t> requested;
  if (auto indices=forward->getAttrOfType<ArrayAttr>("tessera.autodiff.wrt_indices"))
    for (auto index:indices) requested.push_back(cast<IntegerAttr>(index).getInt());
  else requested={0,1,2};
  SmallVector<int64_t> gradientIds(3);
  gradientIds[roles[0]]=rows?12:11;gradientIds[roles[1]]=rows?10:9;
  gradientIds[roles[2]]=rows?11:10;
  SmallVector<int64_t> selected;SmallVector<Type> rootResults;
  for (int64_t index:requested) {selected.push_back(gradientIds[index]);rootResults.push_back(forward.getArgument(index).getType());}
  SmallVector<Type> rootInputs(forward.getArgumentTypes());
  rootInputs.push_back(forward.getResultTypes()[0]);rootInputs.push_back(forward.getResultTypes()[1]);
  builder.setInsertionPointToEnd(module.getBody());
  auto root=func::FuncOp::create(builder,module.getLoc(),rootName,builder.getFunctionType(rootInputs,rootResults));
  auto block=root.addEntryBlock();OpBuilder body(block,block->begin());
  SmallVector<Value> values(block->getArguments());
  SmallVector<SmallVector<int64_t>> inputs{{roles[0]},{5,roles[1],roles[2]},
      {outputSeed,5,roles[1],roles[2],6,7,lseSeed},{roles[0],8}};
  SmallVector<SmallVector<int64_t>> outputs{{5},{6,7},{8,9,10},{11}};
  if (rows) {
    inputs={{roles[0]},{5,roles[1],roles[2]},
        {outputSeed,5,roles[1],roles[2],7,lseSeed},
        {outputSeed,5,roles[1],roles[2],7,8},{roles[0],9}};
    outputs={{5},{6,7},{8},{9,10,11},{12}};
  }
  llvm::json::Array steps;
  SmallVector<int64_t> firstWrite(rows?13:12,-1),lastRead(rows?13:12,-1);
  for (unsigned index=0;index<memberCount;++index) {
    SmallVector<Value> operands;llvm::json::Array inputIds,outputIds;
    for (int64_t id:inputs[index]) {operands.push_back(values[id]);inputIds.push_back(id);lastRead[id]=index;}
    auto call=func::CallOp::create(body,module.getLoc(),members[index],operands);
    for (auto [id,result]:llvm::zip(outputs[index],call.getResults())) {
      if (id!=static_cast<int64_t>(values.size())) return reject();
      values.push_back(result);outputIds.push_back(id);firstWrite[id]=lastRead[id]=index;
    }
    steps.push_back(llvm::json::Object{{"step",int64_t(index)},{"member",members[index].getSymName().str()},
      {"operation",index==0?"tessera.rmsnorm":index==1?"tessera_attn.checkpoint_forward":
                   rows && index==2?"tessera_attn.checkpoint_row_correction":
                   rows && index==3?"tessera_attn.checkpoint_backward_rows":
                   !rows && index==2?"tessera_attn.checkpoint_backward":"native_row_region"},
      {"inputs",std::move(inputIds)},{"outputs",std::move(outputIds)}});
  }
  SmallVector<Value> returned;llvm::json::Array selectedJson,requestedJson,roleJson,seedJson,buffers;
  for (int64_t id:selected) {returned.push_back(values[id]);selectedJson.push_back(id);lastRead[id]=memberCount;}
  for (int64_t id:requested) requestedJson.push_back(id);
  for (int64_t id:roles) roleJson.push_back(id);
  seedJson.push_back(outputSeed);seedJson.push_back(lseSeed);
  func::ReturnOp::create(body,module.getLoc(),returned);
  if (failed(verify(root))) return failure();
  for (unsigned i=0;i<values.size();++i) {
    auto type=cast<RankedTensorType>(values[i].getType());int64_t bytes=type.getElementTypeBitWidth()/8;
    llvm::json::Array shape;
    for (int64_t extent:type.getShape()) {bytes*=extent;shape.push_back(extent);}
    std::string storage;llvm::raw_string_ostream stream(storage);type.getElementType().print(stream);stream.flush();
    bool publicOutput=llvm::is_contained(selected,int64_t(i));
    llvm::json::Object buffer{{"id",int64_t(i)},{"shape",std::move(shape)},{"storage",storage},
      {"bytes",bytes},{"ownership",i<5?"readonly_input":publicOutput?"returned_output":"private_scratch"},
      {"first_write",firstWrite[i]},{"last_read",lastRead[i]}};
    if (bounds) {
      // Root and gradient role IDs survive argument/result permutations.
      bool keySequence=int64_t(i)==roles[1] || int64_t(i)==roles[2] ||
          int64_t(i)==gradientIds[roles[1]] || int64_t(i)==gradientIds[roles[2]];
      llvm::json::Array logical,capacity,trace;
      int64_t capacityBytes=type.getElementTypeBitWidth()/8;
      for (auto [axis,extent]:llvm::enumerate(type.getShape())) {
        trace.push_back(extent);
        int64_t bound=axis==2?bounds[keySequence?4:3]:extent;
        logical.push_back(axis==2?ShapedType::kDynamic:extent);
        capacity.push_back(bound);capacityBytes*=bound;
      }
      buffer["shape"]=std::move(logical);
      buffer["trace_shape"]=std::move(trace);
      buffer["capacity_shape"]=std::move(capacity);
      buffer["bytes"]=capacityBytes;
    }
    buffers.push_back(std::move(buffer));
  }
  auto program=ModuleOp::create(module.getLoc());program->setAttr("tessera.target",target);program->setAttr("tessera.arch",arch);
  program.getBody()->push_back(root->clone());
  for (auto member:members) program.getBody()->push_back(member->clone());
  if (bounds && failed(projectBoundedProgram(program,rootName,bounds,rows,graphs))) {
    program.erase();return failure();
  }
  if (failed(verify(program))) {program.erase();return failure();}
  std::string programText=print(program);program.erase();
  llvm::json::Object plan{{"schema",rows?"tessera.native.source_attention_reverse.v2":
                                             "tessera.native.source_attention_reverse.v1"},
    {"kind",rows?"norm_attention_saved_reverse_rows":"norm_attention_saved_reverse"},{"execution_state","artifact_only"},
    {"target",target.getValue().str()},{"architecture",arch.getValue().str()},
    {"root",rootName},{"source_root",forward.getSymName().str()},
    {"source_graph_ir",originalSource.str()},{"source_graph_sha256",hash(originalSource)},
    {"paired_graph_ir",paired},{"paired_graph_sha256",hash(paired)},
    {"program_graph_ir",programText},{"program_graph_sha256",hash(programText)},
    {"role_indices",std::move(roleJson)},{"cotangent_indices",std::move(seedJson)},
    {"requested_gradients",std::move(requestedJson)},{"buffers",std::move(buffers)},
    {"steps",std::move(steps)},{"outputs",std::move(selectedJson)},{"member_graphs",std::move(graphs)}};
  if (bounds) {
    plan["schema"]=rows?"tessera.native.source_attention_reverse.v4":
                        "tessera.native.source_attention_reverse.v3";
    plan["kind"]=rows?"norm_attention_saved_reverse_rows_bounded":
                      "norm_attention_saved_reverse_bounded";
    plan["shape_policy"]="bounded_sequences_v1";
    plan["sequence_bounds"]=llvm::json::Array{sequence[0],sequence[1]};
    llvm::json::Array capacities;
    for (int64_t extent:bounds.asArrayRef()) capacities.push_back(extent);
    plan["shape_bounds"]=std::move(capacities);
  }
  std::string encoded;llvm::raw_string_ostream stream(encoded);stream<<llvm::json::Value(std::move(plan));stream.flush();
  module->setAttr("tessera.native.source_attention_reverse_program_json",builder.getStringAttr(llvm::encodeBase64(encoded)));
  return success();
}
} // namespace tessera::source_attention_reverse
